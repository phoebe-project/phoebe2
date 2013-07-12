serializeList = function(list) {
    var items = list.getElementsByTagName("li")
    var array = new Array()
    for (var i = 0, n = items.length; i < n; i++) {
        var item = items[i]
        array.push(item.getAttribute("id"))
    }
    return array
};

loop = function(obj, nchild, sel, ul, cl, createpars, mode, selcolor) {
    $.each(obj, function(index, item){
        //~ console.log(index,item,$.type(item))
        if($.type(item)=='array') {
            var ul2 = $("<ul>").appendTo(ul);
            loop(item, nchild[index], sel[index], ul2, cl, createpars, mode, selcolor);
        } else {
            if (item.indexOf('DROP')!=-1) {
                $("<li class='ui-widget-content drop'>", {
                    id: item
                }).text('drop').appendTo(ul);
            } else {
                if (createpars) {
                    $("<li class='ui-widget-content drop parent'>", {
                        id: item
                    }).text("drop parent").appendTo(ul);
                }
                if (mode=='select' & sel[index]){
                    $("<li class='ui-widget-content item "+cl+"'>", {
                        id: item
                    }).text(item).data('maxChildren',nchild[index]).addClass('ui-selected').css('background',selcolor).appendTo(ul);
                    
                } else {
                    $("<li class='ui-widget-content item "+cl+"'>", {
                        id: item
                    }).text(item).data('maxChildren',nchild[index]).appendTo(ul);
                }
            }
        }
    });
};

onDragStart = function(event, ui) {
    var dragitem = ui.helper.prevObject;
    var maxChildren = dragitem.data('maxChildren');
    var minChildren = dragitem.data('minChildren');
    if (dragitem.hasClass('new')!=true) { //then we want to turn the original item into a dropzone
        dragitem.addClass('drop');
    }
    if (maxChildren != '0') {  //then we want to show parents and bump other items to the right temporarily
        $('#groupsys').find('li').addClass('bumpright');
        $('li.parent').show().removeClass('bumpright');
    }
    if (dragitem.hasClass('sys')) { //then we want to show the pool dropzone
        $('#grouppool').find('li.drop').show();
    }
};

onDragStop = function(event, ui){
    console.log('onDragStop')
    ui.helper.prevObject.removeClass('drop'); // restore state of original item
    $('#groupsys').find('li.bumpright').removeClass('bumpright')
    $('li.parent').hide();
    $('#grouppool').find('li.drop').hide();
};

createDropPool = function() {
    $("<li class='ui-widget-content drop pool'>", {
                id: 'droppool'
            })
                .text("drop")
                .droppable( {
                    accept: '#groupsys li',
                    hoverClass: 'hovered',
                    drop: handlePoolDrop
                })
                .appendTo('#grouppool')
                .hide(); 
};

handlePoolDrop = function(event, ui ) {
    messenger.printToTerm('Pool Drop');
    //~ messenger.sendSignal();
    console.log('handlePoolDrop');
    //ui.draggable.draggable( 'disable' );
    //$(this).droppable( 'disable' );
    //ui.draggable.position( { of: $(this), my: 'left top', at: 'left top' } );
    
    ui.draggable.draggable( 'option', 'revert', false );
    var li = ui.helper.prevObject;
    $(this).replaceWith(li).fadeIn();

    // recreate dropzone
    createDropPool();

    //all children of item have to come as well and all get appended to list before last item
    
    //old position needs to be intelligently filled with dropzones

};

handleSysDrop = function(event, ui ) {
    messenger.printToTerm('System Drop');
    console.log('handleSysDrop');
};


var PHOEBE = {
    reset : function (mode, sysitems, sysitems_nchild, font, bgcolor, selcolor) {
        console.log(mode);
        
        if (font) {
            $('html').css('font-family', font);
        }
        
        if (!bgcolor) {
            bgcolor = '#f2f1f0';
        }
        hlcolor = '#dbe5ee';
        
        if (!selcolor) {
            selcolor = '#aaa';
        }
        
        
        $('html').css('background', bgcolor);

        // Reset the items
        // this will eventually be sent from python whenever a change is detected from the backend
        if (mode=='select') {
            $('#groupsys').html( '' );
        } else {
            $('#groupsys').html( '<h2>System</h2>' );
        }

        // Create all lists - eventually sent from python backend and parsed here
        //~ var newitems = {'BinaryStar': '0','BinaryRocheStar': '0', 'Planet': '0', 'Disk': '0', 'Ring': '0', 'BodyBag': '2','NBodyBag': 'N'};
        //~ var sysitems = {'ABC': '2', 'child_ABC': {'StarA': '0', 'BC': '2', 'child_BC': {'StarB': '0', 'CD': '2', 'child_SC': {'DROP1': '0', 'DROP2': '0'}}}};
        //~ var poolitems = {'PoolStar': '0'};
        
        var newitems = ['BinaryStar', 'BinaryRocheStar', 'Planet', 'Disk', 'Ring', 'BodyBag', 'NBodyBag'];
        var newitems_nchild = ['0','0','0','0','0','2','n'];
        var newitems_sel = [false, false, false, false, false, false, false];
        var poolitems = ['PoolStar'];
        var poolitems_nchild = ['0'];
        var poolitems_sel = [false];
        
        if (!sysitems) {
            var sysitems = ['System', ['Primary', 'Secondary']];
            var sysitems_nchild = ['2', ['0','0']];
            var sysitems_sel = [false, [false, false]];
        }
        if (sysitems == 'from_messenger') {
            //~ messenger.printToTerm('from messenger try');
            //~ messenger.printToTerm(messenger.get_test);
            sysitems = JSON.parse(messenger.get_sysitems);
            sysitems_nchild = JSON.parse(messenger.get_sysitems_nchild);
            sysitems_sel = JSON.parse(messenger.get_sysitems_sel);
            //~ messenger.printToTerm(sysitems);
            //~ messenger.printToTerm(sysitems_nchild);
            //~ messenger.printToTerm('after');
        }
        
        // Create nested structures
        ul = $("#groupsys");
        loop(sysitems, sysitems_nchild, sysitems_sel, ul, 'sys', true, mode, selcolor);

        // hide parent items until dragging
        $("li.parent").hide();


        if (mode=='edit') {
            console.log('editmode');
        
            $('#groupnew').html( '<h2>Available</h2>' );
            $('#grouppool').html( '<h2>Pool</h2>' );

            var ul = $("#groupnew");
            loop(newitems, newitems_nchild, newitems_sel, ul, 'new', false, mode);

            
            ul = $("#grouppool");
            loop(poolitems, poolitems_nchild, poolitems_sel, ul, 'pool', false, mode);

            //create pool drop item
            createDropPool();
            
            // make new items draggable
            $("#groupnew").find('li')
                .draggable( {
                    connectToSortable: "#groupsys",
                    //containment: '#body',
                    //stack: '#groupnew div',
                    cursor: 'move',
                    helper: "clone",
                    revert: "invalid",
                    revertDuration: 200,
                    //snap: '.ui-droppable',
                    start: onDragStart,
                    stop: onDragStop
                })
                .disableSelection();
                    
            // make system items draggable
            $("#groupsys, #grouppool").find('li.item')
                .draggable( {
                    //containment: '#content',
                    cursor: 'move',
                    helper: 'clone',
                    revert: true,
                    revertDuration: 200,
                    //snap: '.ui-droppable',
                    start: onDragStart,
                    stop: onDragStop
                })
                .disableSelection();
                
            // enable dropzones
            $("#groupsys").find('li.drop')
                .droppable( {
                    accept: '#groupnew li, #groupsys li, #grouppool li',
                    hoverClass: 'hovered',
                    drop: handleSysDrop
                })
                .disableSelection();
            $("#grouppool").find('li.drop')
                .droppable( {
                    accept: '#groupsys li',
                    hoverClass: 'hovered',
                    drop: handlePoolDrop
                });
        } //end mode edit
        
        if (mode=='select') {
            console.log('selectmode');  
            $('li.drop').hide();
            $('li.parent').hide();          
            $("#body").click(function (event) {
                if ($(event.target).is('ul')) { //then clicking on background and not just element
                        messenger.sendSignal('editClicked');
                }
            });
                
            $("#body").mouseover(function (event) {
                if ($(event.target).is('ul')) { //then clicking on background and not just element
                        $("html").css("background-color", hlcolor);
                    } else {
                        $("html").css("background-color", bgcolor);
                        
                }
            });
            
            $("#groupsys").css("height","80%");
            
            $("#body").mouseout(function (event) {
                $("html").css("background-color", bgcolor);
            });
                
            $("#groupsys").on('click', 'li', function (e) {
                if (messenger.get_ctrl=='True') {
                    $(this).addClass("ui-selected").css('background',selcolor);  //should be toggleClass but this is getting called twice for some reason so that isn't working
                } else {
                    $(this).addClass("ui-selected").css('background',selcolor).siblings().removeClass('ui-selected').css('background','#fff');
                    $("#groupsys").find('li').removeClass('ui-selected').css('background','#fff');
                    $(this).addClass("ui-selected").css('background',selcolor);
                }
                // update selected values, through messenger to gui
                var selected = $.map(sysitems_sel, function(n){return n;});
                $.each($("#groupsys").find('li.item'), function(index, item) {
                    selected[index] = $(this).hasClass('ui-selected');
                });
                messenger.updateSelected(JSON.stringify(selected));
                //~ messenger.printToTerm(JSON.sringify(selected));
                //~ $('#groupsys').css('background','#fff');
                //~ $('.ui-selected').css('background',selcolor);
            });

            //~ $("#groupsys")
                //~ .selectable( {
                    //~ stop: function () {
                        //~ console.log('select stop')
                    //~ }
                //~ });
            
            //color on main div hover
            
            //make click on main div send signal

            
            
        } //end mode select
    } //end reset
}; //end PHOEBE
